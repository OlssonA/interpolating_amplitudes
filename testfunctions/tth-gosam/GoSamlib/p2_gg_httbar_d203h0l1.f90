module     p2_gg_httbar_d203h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d203h0l1.f90
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
      use p2_gg_httbar_abbrevd203h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc203(39)
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae1e2
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      acc203(1)=abb203(45)
      acc203(2)=abb203(46)
      acc203(3)=abb203(47)
      acc203(4)=abb203(48)
      acc203(5)=abb203(49)
      acc203(6)=abb203(50)
      acc203(7)=abb203(52)
      acc203(8)=abb203(53)
      acc203(9)=abb203(54)
      acc203(10)=abb203(55)
      acc203(11)=abb203(56)
      acc203(12)=abb203(57)
      acc203(13)=abb203(58)
      acc203(14)=abb203(59)
      acc203(15)=abb203(60)
      acc203(16)=abb203(61)
      acc203(17)=abb203(62)
      acc203(18)=abb203(63)
      acc203(19)=abb203(64)
      acc203(20)=abb203(65)
      acc203(21)=abb203(66)
      acc203(22)=abb203(67)
      acc203(23)=abb203(68)
      acc203(24)=abb203(69)
      acc203(25)=abb203(71)
      acc203(26)=abb203(72)
      acc203(27)=abb203(73)
      acc203(28)=Qspval3e2*acc203(3)
      acc203(29)=Qspval4e2*acc203(23)
      acc203(30)=Qspval5e2*acc203(26)
      acc203(28)=acc203(30)+acc203(29)-acc203(1)+acc203(28)
      acc203(28)=Qspvae1k2*acc203(28)
      acc203(29)=Qspval3e2*acc203(13)
      acc203(30)=-Qspvae1l3*acc203(12)
      acc203(31)=Qspvae1l3*acc203(5)
      acc203(31)=acc203(22)+acc203(31)
      acc203(31)=Qspval4e2*acc203(31)
      acc203(32)=Qspvae1l3*acc203(7)
      acc203(32)=acc203(19)+acc203(32)
      acc203(32)=Qspval5e2*acc203(32)
      acc203(28)=acc203(28)+acc203(32)+acc203(31)+acc203(30)+acc203(8)+acc203(2&
      &9)
      acc203(28)=Qspvae2e1*acc203(28)
      acc203(29)=acc203(3)*Qspval3e1
      acc203(30)=Qspval4e1*acc203(23)
      acc203(31)=Qspval5e1*acc203(26)
      acc203(29)=acc203(31)+acc203(30)+acc203(27)+acc203(29)
      acc203(29)=Qspvae2k2*acc203(29)
      acc203(30)=Qspvae2l3*acc203(16)
      acc203(31)=Qspvae2l3*acc203(5)
      acc203(31)=acc203(20)+acc203(31)
      acc203(31)=Qspval4e1*acc203(31)
      acc203(32)=Qspvae2l3*acc203(7)
      acc203(32)=acc203(21)+acc203(32)
      acc203(32)=Qspval5e1*acc203(32)
      acc203(29)=acc203(29)+acc203(32)+acc203(31)+acc203(10)+acc203(30)
      acc203(29)=Qspvae1e2*acc203(29)
      acc203(30)=Qspval3e1*acc203(14)
      acc203(31)=Qspval3e2*acc203(17)
      acc203(32)=Qspvae1l3*acc203(15)
      acc203(33)=Qspvae2l3*acc203(6)
      acc203(34)=-Qspval4e1*acc203(4)
      acc203(35)=Qspval4e2*acc203(9)
      acc203(36)=-Qspval5e1*acc203(25)
      acc203(37)=Qspval5e2*acc203(18)
      acc203(38)=Qspvae1k2*acc203(11)
      acc203(39)=Qspvae2k2*acc203(24)
      brack=acc203(2)+acc203(28)+acc203(29)+acc203(30)+acc203(31)+acc203(32)+ac&
      &c203(33)+acc203(34)+acc203(35)+acc203(36)+acc203(37)+acc203(38)+acc203(3&
      &9)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d203h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd203h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d203
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d203 = 0.0_ki
      d203 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d203, ki), aimag(d203), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d203h0l1
