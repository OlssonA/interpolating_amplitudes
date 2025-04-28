module     p2_gg_httbar_d89h12l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d89h12l1.f90
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
      use p2_gg_httbar_abbrevd89h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc89(46)
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc89(1)=abb89(8)
      acc89(2)=abb89(9)
      acc89(3)=abb89(10)
      acc89(4)=abb89(11)
      acc89(5)=abb89(12)
      acc89(6)=abb89(13)
      acc89(7)=abb89(14)
      acc89(8)=abb89(15)
      acc89(9)=abb89(16)
      acc89(10)=abb89(17)
      acc89(11)=abb89(18)
      acc89(12)=abb89(19)
      acc89(13)=abb89(20)
      acc89(14)=abb89(21)
      acc89(15)=abb89(23)
      acc89(16)=abb89(24)
      acc89(17)=abb89(25)
      acc89(18)=abb89(26)
      acc89(19)=abb89(27)
      acc89(20)=abb89(28)
      acc89(21)=abb89(30)
      acc89(22)=abb89(31)
      acc89(23)=abb89(32)
      acc89(24)=abb89(33)
      acc89(25)=abb89(35)
      acc89(26)=abb89(36)
      acc89(27)=abb89(38)
      acc89(28)=abb89(46)
      acc89(29)=abb89(54)
      acc89(30)=-acc89(16)*Qspvae2l5
      acc89(31)=acc89(21)*Qspvae2l3
      acc89(30)=acc89(31)+acc89(30)+acc89(12)
      acc89(30)=acc89(30)*Qspvak2e1
      acc89(31)=acc89(5)*Qspvae2l5
      acc89(32)=acc89(10)*Qspvae2l3
      acc89(30)=acc89(22)+acc89(32)+acc89(31)+acc89(30)
      acc89(30)=Qspvae1e2*acc89(30)
      acc89(31)=-acc89(2)*Qspvak2e2
      acc89(32)=acc89(26)*Qspval3e2
      acc89(31)=acc89(32)+acc89(14)+acc89(31)
      acc89(31)=acc89(31)*Qspvae1l4
      acc89(32)=acc89(4)*Qspvak2e2
      acc89(33)=acc89(25)*Qspval3e2
      acc89(31)=acc89(33)+acc89(32)+acc89(1)+acc89(31)
      acc89(31)=Qspvae2e1*acc89(31)
      acc89(32)=acc89(18)*Qspvae1e2
      acc89(32)=acc89(27)+acc89(32)
      acc89(32)=Qspvae2l4*acc89(32)
      acc89(33)=acc89(3)*Qspvak2e2
      acc89(34)=acc89(6)*Qspvak2e1
      acc89(35)=acc89(9)*Qspvae2l3
      acc89(36)=acc89(13)*Qspval3e2
      acc89(37)=acc89(24)*Qspvae1l4
      acc89(38)=acc89(28)*Qspvae2l5
      acc89(39)=Qspvae2k1*acc89(23)
      acc89(40)=Qspvak1e2*acc89(19)
      acc89(41)=-Qspval3k2*acc89(29)
      acc89(42)=Qspvak2l5*acc89(7)
      acc89(43)=-Qspvak2l4*acc89(20)
      acc89(44)=Qspvak2l3*acc89(11)
      acc89(45)=Qspk2*acc89(15)
      acc89(46)=QspQ*acc89(17)
      brack=acc89(8)+acc89(30)+acc89(31)+acc89(32)+acc89(33)+acc89(34)+acc89(35&
      &)+acc89(36)+acc89(37)+acc89(38)+acc89(39)+acc89(40)+acc89(41)+acc89(42)+&
      &acc89(43)+acc89(44)+acc89(45)+acc89(46)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d89h12l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd89h12
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d89
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d89 = 0.0_ki
      d89 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d89, ki), aimag(d89), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d89h12l1
