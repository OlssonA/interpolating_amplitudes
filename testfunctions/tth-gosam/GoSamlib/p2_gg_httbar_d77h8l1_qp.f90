module     p2_gg_httbar_d77h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d77h8l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd77h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc77(103)
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspk2
      complex(ki) :: Qspl5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak2l5
      complex(ki) :: QspQ
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak2k1
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspk2 = dotproduct(Q,k2)
      Qspl5 = dotproduct(Q,l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      QspQ = dotproduct(Q,Q)
      Qspe1 = dotproduct(Q,e1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      acc77(1)=abb77(9)
      acc77(2)=abb77(10)
      acc77(3)=abb77(11)
      acc77(4)=abb77(12)
      acc77(5)=abb77(13)
      acc77(6)=abb77(14)
      acc77(7)=abb77(15)
      acc77(8)=abb77(16)
      acc77(9)=abb77(17)
      acc77(10)=abb77(18)
      acc77(11)=abb77(19)
      acc77(12)=abb77(20)
      acc77(13)=abb77(21)
      acc77(14)=abb77(22)
      acc77(15)=abb77(23)
      acc77(16)=abb77(24)
      acc77(17)=abb77(25)
      acc77(18)=abb77(26)
      acc77(19)=abb77(27)
      acc77(20)=abb77(28)
      acc77(21)=abb77(29)
      acc77(22)=abb77(30)
      acc77(23)=abb77(31)
      acc77(24)=abb77(32)
      acc77(25)=abb77(33)
      acc77(26)=abb77(34)
      acc77(27)=abb77(35)
      acc77(28)=abb77(36)
      acc77(29)=abb77(37)
      acc77(30)=abb77(38)
      acc77(31)=abb77(39)
      acc77(32)=abb77(40)
      acc77(33)=abb77(41)
      acc77(34)=abb77(42)
      acc77(35)=abb77(43)
      acc77(36)=abb77(44)
      acc77(37)=abb77(45)
      acc77(38)=abb77(46)
      acc77(39)=abb77(47)
      acc77(40)=abb77(48)
      acc77(41)=abb77(49)
      acc77(42)=abb77(50)
      acc77(43)=abb77(53)
      acc77(44)=abb77(54)
      acc77(45)=abb77(55)
      acc77(46)=abb77(56)
      acc77(47)=abb77(58)
      acc77(48)=abb77(60)
      acc77(49)=abb77(62)
      acc77(50)=abb77(63)
      acc77(51)=abb77(64)
      acc77(52)=abb77(65)
      acc77(53)=abb77(68)
      acc77(54)=abb77(73)
      acc77(55)=abb77(74)
      acc77(56)=abb77(82)
      acc77(57)=abb77(84)
      acc77(58)=abb77(86)
      acc77(59)=abb77(87)
      acc77(60)=abb77(90)
      acc77(61)=abb77(91)
      acc77(62)=abb77(114)
      acc77(63)=abb77(129)
      acc77(64)=abb77(132)
      acc77(65)=abb77(135)
      acc77(66)=abb77(153)
      acc77(67)=Qspval3l5*acc77(20)
      acc77(68)=Qspk2*acc77(2)
      acc77(69)=Qspl5*acc77(61)
      acc77(70)=Qspvak2l3*acc77(32)
      acc77(71)=Qspvak2l4*acc77(40)
      acc77(72)=Qspval3l4*acc77(36)
      acc77(73)=Qspval4l3*acc77(29)
      acc77(74)=Qspval4l5*acc77(27)
      acc77(75)=Qspvak2e2*acc77(56)
      acc77(76)=Qspval3e2*acc77(53)
      acc77(77)=Qspvae2l3*acc77(48)
      acc77(78)=Qspvae2l5*acc77(37)
      acc77(79)=Qspvak2l5*acc77(19)
      acc77(80)=QspQ*acc77(5)
      acc77(67)=acc77(80)+acc77(79)+acc77(78)+acc77(77)+acc77(76)+acc77(75)+acc&
      &77(74)+acc77(73)+acc77(72)+acc77(71)+acc77(70)+acc77(69)+acc77(68)+acc77&
      &(44)+acc77(67)
      acc77(67)=Qspe1*acc77(67)
      acc77(68)=-acc77(64)*Qspvae1l4
      acc77(69)=-acc77(63)*Qspval4e1
      acc77(70)=acc77(47)*Qspvae2e1
      acc77(71)=-acc77(23)*Qspvae1e2
      acc77(72)=-Qspvak2e1*acc77(16)
      acc77(73)=Qspvae1l5*acc77(33)
      acc77(68)=acc77(73)+acc77(72)+acc77(71)+acc77(70)+acc77(69)+acc77(4)+acc7&
      &7(68)
      acc77(68)=QspQ*acc77(68)
      acc77(69)=Qspvak2l4*acc77(28)
      acc77(70)=Qspvak2e2*acc77(55)
      acc77(71)=Qspvak2l5*acc77(6)
      acc77(69)=acc77(71)+acc77(70)+acc77(50)+acc77(69)
      acc77(69)=Qspvae1k2*acc77(69)
      acc77(70)=Qspvak2l3*acc77(41)
      acc77(71)=-acc77(63)*Qspval4l3
      acc77(72)=acc77(47)*Qspvae2l3
      acc77(70)=acc77(72)+acc77(71)+acc77(58)+acc77(70)
      acc77(70)=Qspval3e1*acc77(70)
      acc77(71)=-acc77(63)*Qspval4l5
      acc77(72)=acc77(47)*Qspvae2l5
      acc77(73)=-Qspvak2l5*acc77(16)
      acc77(71)=acc77(73)+acc77(72)+acc77(38)+acc77(71)
      acc77(71)=Qspval5e1*acc77(71)
      acc77(72)=Qspval4k2*acc77(11)
      acc77(73)=Qspvae2k2*acc77(8)
      acc77(74)=Qspk2*acc77(10)
      acc77(72)=acc77(74)+acc77(73)+acc77(12)+acc77(72)
      acc77(72)=Qspvak2e1*acc77(72)
      acc77(73)=Qspl5*acc77(46)
      acc77(74)=-acc77(64)*Qspval5l4
      acc77(75)=-acc77(23)*Qspval5e2
      acc77(73)=acc77(75)+acc77(74)+acc77(3)+acc77(73)
      acc77(73)=Qspvae1l5*acc77(73)
      acc77(74)=-acc77(64)*Qspval3l4
      acc77(75)=-acc77(23)*Qspval3e2
      acc77(74)=acc77(75)+acc77(74)+acc77(54)
      acc77(74)=Qspvae1l3*acc77(74)
      acc77(75)=acc77(39)*Qspval3k1
      acc77(76)=acc77(34)*Qspvak1k2
      acc77(77)=acc77(26)*Qspvak1l3
      acc77(78)=acc77(22)*Qspval5k1
      acc77(79)=acc77(21)*Qspvak1e1
      acc77(80)=acc77(18)*Qspvae1k1
      acc77(81)=acc77(15)*Qspvak1l5
      acc77(82)=acc77(9)*Qspvak2k1
      acc77(83)=Qspval3l5*acc77(30)
      acc77(84)=-Qspval4k2*acc77(51)
      acc77(85)=Qspval5l4*acc77(65)
      acc77(86)=Qspvae2k2*acc77(45)
      acc77(87)=Qspval4e1*acc77(43)
      acc77(88)=Qspvae1l4*acc77(42)
      acc77(89)=Qspval5e2*acc77(66)
      acc77(90)=Qspvae1e2*acc77(13)
      acc77(91)=Qspvae2e1*acc77(7)
      acc77(92)=Qspk2*acc77(1)
      acc77(93)=Qspl5*acc77(52)
      acc77(94)=Qspvak2l3*acc77(25)
      acc77(95)=Qspvak2l4*acc77(31)
      acc77(96)=Qspval3l4*acc77(35)
      acc77(97)=-Qspval4l3*acc77(59)
      acc77(98)=Qspval4l5*acc77(62)
      acc77(99)=Qspvak2e2*acc77(60)
      acc77(100)=Qspval3e2*acc77(49)
      acc77(101)=Qspvae2l3*acc77(57)
      acc77(102)=Qspvae2l5*acc77(24)
      acc77(103)=Qspvak2l5*acc77(14)
      brack=acc77(17)+acc77(67)+acc77(68)+acc77(69)+acc77(70)+acc77(71)+acc77(7&
      &2)+acc77(73)+acc77(74)+acc77(75)+acc77(76)+acc77(77)+acc77(78)+acc77(79)&
      &+acc77(80)+acc77(81)+acc77(82)+acc77(83)+acc77(84)+acc77(85)+acc77(86)+a&
      &cc77(87)+acc77(88)+acc77(89)+acc77(90)+acc77(91)+acc77(92)+acc77(93)+acc&
      &77(94)+acc77(95)+acc77(96)+acc77(97)+acc77(98)+acc77(99)+acc77(100)+acc7&
      &7(101)+acc77(102)+acc77(103)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d77h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd77h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d77
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d77 = 0.0_ki
      d77 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d77, ki), aimag(d77), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d77h8l1_qp
