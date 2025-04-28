module     p2_gg_httbar_d27h12l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d27h12l1.f90
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
      use p2_gg_httbar_abbrevd27h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc27(37)
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2l3
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      acc27(1)=abb27(10)
      acc27(2)=abb27(11)
      acc27(3)=abb27(12)
      acc27(4)=abb27(13)
      acc27(5)=abb27(14)
      acc27(6)=abb27(15)
      acc27(7)=abb27(16)
      acc27(8)=abb27(18)
      acc27(9)=abb27(19)
      acc27(10)=abb27(20)
      acc27(11)=abb27(21)
      acc27(12)=abb27(24)
      acc27(13)=abb27(25)
      acc27(14)=abb27(28)
      acc27(15)=abb27(29)
      acc27(16)=abb27(30)
      acc27(17)=abb27(31)
      acc27(18)=abb27(34)
      acc27(19)=abb27(36)
      acc27(20)=abb27(41)
      acc27(21)=abb27(43)
      acc27(22)=abb27(45)
      acc27(23)=abb27(46)
      acc27(24)=abb27(48)
      acc27(25)=abb27(50)
      acc27(26)=abb27(63)
      acc27(27)=acc27(10)*Qspvae2k1
      acc27(28)=acc27(14)*Qspvae2k2
      acc27(29)=-acc27(18)*Qspvae2e1
      acc27(30)=acc27(24)*Qspvae2l4
      acc27(27)=acc27(30)+acc27(21)+acc27(29)+acc27(28)+acc27(27)
      acc27(27)=Qspval3e2*acc27(27)
      acc27(28)=acc27(6)*Qspvae2e1
      acc27(29)=acc27(9)*Qspvae2k2
      acc27(30)=acc27(11)*Qspvae2k1
      acc27(31)=acc27(12)*Qspvae2l4
      acc27(28)=acc27(31)+acc27(30)+acc27(29)+acc27(28)+acc27(3)
      acc27(28)=Qspvak2e2*acc27(28)
      acc27(29)=acc27(17)*Qspval4e2
      acc27(30)=acc27(23)*Qspvae1e2
      acc27(31)=-acc27(26)*Qspvak1e2
      acc27(29)=acc27(31)+acc27(30)+acc27(29)+acc27(2)
      acc27(29)=Qspvae2l5*acc27(29)
      acc27(30)=acc27(19)*Qspval4e2
      acc27(31)=-acc27(22)*Qspvak1e2
      acc27(32)=acc27(25)*Qspvae1e2
      acc27(30)=acc27(32)+acc27(31)+acc27(30)+acc27(16)
      acc27(30)=Qspvae2l3*acc27(30)
      acc27(31)=acc27(4)*Qspvae2k1
      acc27(32)=acc27(5)*Qspvae2e1
      acc27(33)=acc27(7)*Qspvak1e2
      acc27(34)=acc27(8)*Qspvae2l4
      acc27(35)=acc27(13)*Qspvae1e2
      acc27(36)=acc27(15)*Qspval4e2
      acc27(37)=acc27(20)*Qspvae2k2
      brack=acc27(1)+acc27(27)+acc27(28)+acc27(29)+acc27(30)+acc27(31)+acc27(32&
      &)+acc27(33)+acc27(34)+acc27(35)+acc27(36)+acc27(37)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d27h12l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd27h12
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d27
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k3+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d27 = 0.0_ki
      d27 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d27, ki), aimag(d27), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d27h12l1
