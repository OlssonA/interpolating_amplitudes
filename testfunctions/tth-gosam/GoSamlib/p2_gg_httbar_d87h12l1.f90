module     p2_gg_httbar_d87h12l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d87h12l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd87h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc87(44)
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l3
      complex(ki) :: QspQ
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      QspQ = dotproduct(Q,Q)
      acc87(1)=abb87(8)
      acc87(2)=abb87(9)
      acc87(3)=abb87(10)
      acc87(4)=abb87(11)
      acc87(5)=abb87(12)
      acc87(6)=abb87(13)
      acc87(7)=abb87(14)
      acc87(8)=abb87(15)
      acc87(9)=abb87(16)
      acc87(10)=abb87(17)
      acc87(11)=abb87(18)
      acc87(12)=abb87(19)
      acc87(13)=abb87(20)
      acc87(14)=abb87(22)
      acc87(15)=abb87(23)
      acc87(16)=abb87(24)
      acc87(17)=abb87(25)
      acc87(18)=abb87(26)
      acc87(19)=abb87(27)
      acc87(20)=abb87(28)
      acc87(21)=abb87(29)
      acc87(22)=abb87(30)
      acc87(23)=abb87(31)
      acc87(24)=abb87(32)
      acc87(25)=abb87(33)
      acc87(26)=abb87(35)
      acc87(27)=abb87(37)
      acc87(28)=abb87(39)
      acc87(29)=acc87(2)*Qspvae1l5
      acc87(30)=acc87(6)*Qspvae1l3
      acc87(29)=acc87(18)+acc87(30)+acc87(29)
      acc87(29)=acc87(29)*Qspvak2e2
      acc87(30)=acc87(23)*Qspvae1l3
      acc87(31)=acc87(26)*Qspvae1l5
      acc87(29)=acc87(31)+acc87(30)+acc87(10)+acc87(29)
      acc87(29)=Qspvae2e1*acc87(29)
      acc87(30)=-acc87(20)*Qspvak2e1
      acc87(31)=acc87(28)*Qspval3e1
      acc87(30)=acc87(31)+acc87(30)+acc87(15)
      acc87(30)=acc87(30)*Qspvae2l4
      acc87(31)=acc87(5)*Qspvak2e1
      acc87(32)=acc87(24)*Qspval3e1
      acc87(30)=acc87(32)+acc87(17)+acc87(31)+acc87(30)
      acc87(30)=Qspvae1e2*acc87(30)
      acc87(31)=acc87(14)*Qspvae2e1
      acc87(31)=acc87(19)+acc87(31)
      acc87(31)=Qspvae1l4*acc87(31)
      acc87(32)=acc87(1)*Qspvak2e2
      acc87(33)=acc87(8)*Qspvak2e1
      acc87(34)=acc87(11)*Qspvae2l4
      acc87(35)=acc87(12)*Qspval3e1
      acc87(36)=acc87(21)*Qspvae1l5
      acc87(37)=acc87(22)*Qspvae1l3
      acc87(38)=-Qspvae1k2*acc87(27)
      acc87(39)=Qspval3k1*acc87(25)
      acc87(40)=Qspvak2k1*acc87(13)
      acc87(41)=Qspvak1l5*acc87(3)
      acc87(42)=Qspvak1l4*acc87(16)
      acc87(43)=Qspvak1l3*acc87(7)
      acc87(44)=QspQ*acc87(9)
      brack=acc87(4)+acc87(29)+acc87(30)+acc87(31)+acc87(32)+acc87(33)+acc87(34&
      &)+acc87(35)+acc87(36)+acc87(37)+acc87(38)+acc87(39)+acc87(40)+acc87(41)+&
      &acc87(42)+acc87(43)+acc87(44)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d87h12l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd87h12
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d87
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d87 = 0.0_ki
      d87 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d87, ki), aimag(d87), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d87h12l1
