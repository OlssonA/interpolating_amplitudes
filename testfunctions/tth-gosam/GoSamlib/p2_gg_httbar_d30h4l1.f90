module     p2_gg_httbar_d30h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d30h4l1.f90
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
      use p2_gg_httbar_abbrevd30h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc30(35)
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval3e2
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      acc30(1)=abb30(10)
      acc30(2)=abb30(11)
      acc30(3)=abb30(12)
      acc30(4)=abb30(13)
      acc30(5)=abb30(14)
      acc30(6)=abb30(15)
      acc30(7)=abb30(16)
      acc30(8)=abb30(17)
      acc30(9)=abb30(18)
      acc30(10)=abb30(19)
      acc30(11)=abb30(20)
      acc30(12)=abb30(21)
      acc30(13)=abb30(22)
      acc30(14)=abb30(23)
      acc30(15)=abb30(24)
      acc30(16)=abb30(33)
      acc30(17)=abb30(34)
      acc30(18)=abb30(35)
      acc30(19)=abb30(38)
      acc30(20)=abb30(39)
      acc30(21)=abb30(40)
      acc30(22)=abb30(41)
      acc30(23)=abb30(42)
      acc30(24)=abb30(61)
      acc30(25)=abb30(63)
      acc30(26)=acc30(4)*Qspvae2l3
      acc30(27)=acc30(7)*Qspvae2k1
      acc30(28)=acc30(10)*Qspvae2e1
      acc30(29)=acc30(11)*Qspvae2l4
      acc30(30)=acc30(13)*Qspvae2l5
      acc30(26)=acc30(30)+acc30(29)+acc30(28)+acc30(27)+acc30(26)+acc30(1)
      acc30(26)=Qspvak2e2*acc30(26)
      acc30(27)=-acc30(21)*Qspval5e2
      acc30(28)=acc30(22)*Qspvae1e2
      acc30(29)=acc30(23)*Qspvak1e2
      acc30(27)=acc30(29)+acc30(28)+acc30(27)+acc30(20)
      acc30(27)=Qspvae2l4*acc30(27)
      acc30(28)=acc30(19)*Qspvae1e2
      acc30(29)=-acc30(24)*Qspval5e2
      acc30(30)=acc30(25)*Qspvak1e2
      acc30(28)=acc30(30)+acc30(29)+acc30(28)+acc30(14)
      acc30(28)=Qspvae2l3*acc30(28)
      acc30(29)=acc30(6)*Qspvae2k1
      acc30(30)=acc30(16)*Qspvae2e1
      acc30(31)=acc30(18)*Qspvae2l5
      acc30(29)=acc30(31)+acc30(30)+acc30(12)+acc30(29)
      acc30(29)=Qspval3e2*acc30(29)
      acc30(30)=acc30(3)*Qspvae2e1
      acc30(31)=acc30(5)*Qspvae2k1
      acc30(32)=acc30(8)*Qspvae1e2
      acc30(33)=acc30(9)*Qspvae2l5
      acc30(34)=acc30(15)*Qspvak1e2
      acc30(35)=acc30(17)*Qspval5e2
      brack=acc30(2)+acc30(26)+acc30(27)+acc30(28)+acc30(29)+acc30(30)+acc30(31&
      &)+acc30(32)+acc30(33)+acc30(34)+acc30(35)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d30h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd30h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d30
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k3+k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d30 = 0.0_ki
      d30 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d30, ki), aimag(d30), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d30h4l1
