module     p2_gg_httbar_d80h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d80h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd80h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc80(103)
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspk2
      complex(ki) :: Qspl4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspval4k2
      complex(ki) :: QspQ
      complex(ki) :: Qspe1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1k2
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspk2 = dotproduct(Q,k2)
      Qspl4 = dotproduct(Q,l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      QspQ = dotproduct(Q,Q)
      Qspe1 = dotproduct(Q,e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      acc80(1)=abb80(9)
      acc80(2)=abb80(10)
      acc80(3)=abb80(11)
      acc80(4)=abb80(12)
      acc80(5)=abb80(13)
      acc80(6)=abb80(14)
      acc80(7)=abb80(15)
      acc80(8)=abb80(16)
      acc80(9)=abb80(17)
      acc80(10)=abb80(18)
      acc80(11)=abb80(19)
      acc80(12)=abb80(20)
      acc80(13)=abb80(21)
      acc80(14)=abb80(22)
      acc80(15)=abb80(24)
      acc80(16)=abb80(25)
      acc80(17)=abb80(26)
      acc80(18)=abb80(27)
      acc80(19)=abb80(28)
      acc80(20)=abb80(29)
      acc80(21)=abb80(30)
      acc80(22)=abb80(31)
      acc80(23)=abb80(32)
      acc80(24)=abb80(33)
      acc80(25)=abb80(34)
      acc80(26)=abb80(35)
      acc80(27)=abb80(36)
      acc80(28)=abb80(37)
      acc80(29)=abb80(38)
      acc80(30)=abb80(39)
      acc80(31)=abb80(40)
      acc80(32)=abb80(41)
      acc80(33)=abb80(43)
      acc80(34)=abb80(44)
      acc80(35)=abb80(45)
      acc80(36)=abb80(46)
      acc80(37)=abb80(47)
      acc80(38)=abb80(48)
      acc80(39)=abb80(49)
      acc80(40)=abb80(50)
      acc80(41)=abb80(51)
      acc80(42)=abb80(52)
      acc80(43)=abb80(53)
      acc80(44)=abb80(54)
      acc80(45)=abb80(55)
      acc80(46)=abb80(56)
      acc80(47)=abb80(57)
      acc80(48)=abb80(58)
      acc80(49)=abb80(60)
      acc80(50)=abb80(62)
      acc80(51)=abb80(63)
      acc80(52)=abb80(64)
      acc80(53)=abb80(65)
      acc80(54)=abb80(68)
      acc80(55)=abb80(73)
      acc80(56)=abb80(82)
      acc80(57)=abb80(84)
      acc80(58)=abb80(87)
      acc80(59)=abb80(89)
      acc80(60)=abb80(90)
      acc80(61)=abb80(91)
      acc80(62)=abb80(114)
      acc80(63)=abb80(129)
      acc80(64)=abb80(132)
      acc80(65)=abb80(135)
      acc80(66)=abb80(153)
      acc80(67)=Qspval4l3*acc80(19)
      acc80(68)=Qspk2*acc80(2)
      acc80(69)=Qspl4*acc80(61)
      acc80(70)=Qspval3k2*acc80(30)
      acc80(71)=Qspval3l5*acc80(27)
      acc80(72)=Qspval4l5*acc80(51)
      acc80(73)=Qspval5k2*acc80(41)
      acc80(74)=Qspval5l3*acc80(33)
      acc80(75)=Qspvae2k2*acc80(56)
      acc80(76)=Qspval3e2*acc80(54)
      acc80(77)=Qspvae2l3*acc80(50)
      acc80(78)=Qspval4e2*acc80(40)
      acc80(79)=Qspval4k2*acc80(18)
      acc80(80)=QspQ*acc80(5)
      acc80(67)=acc80(80)+acc80(79)+acc80(78)+acc80(77)+acc80(76)+acc80(75)+acc&
      &80(74)+acc80(73)+acc80(72)+acc80(71)+acc80(70)+acc80(69)+acc80(68)+acc80&
      &(44)+acc80(67)
      acc80(67)=Qspe1*acc80(67)
      acc80(68)=acc80(64)*Qspval5e1
      acc80(69)=acc80(63)*Qspvae1l5
      acc80(70)=-acc80(48)*Qspvae1e2
      acc80(71)=acc80(22)*Qspvae2e1
      acc80(72)=Qspvae1k2*acc80(15)
      acc80(73)=Qspval4e1*acc80(32)
      acc80(68)=acc80(73)+acc80(72)+acc80(71)+acc80(70)+acc80(69)+acc80(4)+acc8&
      &0(68)
      acc80(68)=QspQ*acc80(68)
      acc80(69)=Qspval5k2*acc80(31)
      acc80(70)=Qspvae2k2*acc80(20)
      acc80(71)=-Qspval4k2*acc80(6)
      acc80(69)=acc80(71)+acc80(70)+acc80(17)+acc80(69)
      acc80(69)=Qspvak2e1*acc80(69)
      acc80(70)=Qspval3k2*acc80(39)
      acc80(71)=acc80(63)*Qspval3l5
      acc80(72)=-acc80(48)*Qspval3e2
      acc80(70)=acc80(72)+acc80(71)+acc80(55)+acc80(70)
      acc80(70)=Qspvae1l3*acc80(70)
      acc80(71)=acc80(63)*Qspval4l5
      acc80(72)=-acc80(48)*Qspval4e2
      acc80(73)=Qspval4k2*acc80(15)
      acc80(71)=acc80(73)+acc80(72)+acc80(43)+acc80(71)
      acc80(71)=Qspvae1l4*acc80(71)
      acc80(72)=Qspvak2l5*acc80(11)
      acc80(73)=Qspvak2e2*acc80(8)
      acc80(74)=Qspk2*acc80(10)
      acc80(72)=acc80(74)+acc80(73)+acc80(12)+acc80(72)
      acc80(72)=Qspvae1k2*acc80(72)
      acc80(73)=-Qspl4*acc80(46)
      acc80(74)=acc80(64)*Qspval5l4
      acc80(75)=acc80(22)*Qspvae2l4
      acc80(73)=acc80(75)+acc80(74)+acc80(3)+acc80(73)
      acc80(73)=Qspval4e1*acc80(73)
      acc80(74)=acc80(64)*Qspval5l3
      acc80(75)=acc80(22)*Qspvae2l3
      acc80(74)=acc80(75)+acc80(74)+acc80(59)
      acc80(74)=Qspval3e1*acc80(74)
      acc80(75)=acc80(47)*Qspvak2k1
      acc80(76)=acc80(42)*Qspval3k1
      acc80(77)=acc80(38)*Qspvak1l3
      acc80(78)=acc80(37)*Qspvak1l4
      acc80(79)=acc80(26)*Qspvak1e1
      acc80(80)=acc80(25)*Qspval4k1
      acc80(81)=acc80(21)*Qspvae1k1
      acc80(82)=acc80(9)*Qspvak1k2
      acc80(83)=Qspvak2l5*acc80(52)
      acc80(84)=Qspval4l3*acc80(29)
      acc80(85)=-Qspval5l4*acc80(65)
      acc80(86)=-Qspvak2e2*acc80(45)
      acc80(87)=-Qspvae2l4*acc80(66)
      acc80(88)=Qspval5e1*acc80(35)
      acc80(89)=Qspvae1l5*acc80(23)
      acc80(90)=Qspvae1e2*acc80(7)
      acc80(91)=Qspvae2e1*acc80(13)
      acc80(92)=Qspk2*acc80(1)
      acc80(93)=Qspl4*acc80(53)
      acc80(94)=Qspval3k2*acc80(24)
      acc80(95)=Qspval3l5*acc80(58)
      acc80(96)=-Qspval4l5*acc80(62)
      acc80(97)=Qspval5k2*acc80(34)
      acc80(98)=Qspval5l3*acc80(28)
      acc80(99)=Qspvae2k2*acc80(60)
      acc80(100)=-Qspval3e2*acc80(57)
      acc80(101)=Qspvae2l3*acc80(49)
      acc80(102)=Qspval4e2*acc80(36)
      acc80(103)=Qspval4k2*acc80(14)
      brack=acc80(16)+acc80(67)+acc80(68)+acc80(69)+acc80(70)+acc80(71)+acc80(7&
      &2)+acc80(73)+acc80(74)+acc80(75)+acc80(76)+acc80(77)+acc80(78)+acc80(79)&
      &+acc80(80)+acc80(81)+acc80(82)+acc80(83)+acc80(84)+acc80(85)+acc80(86)+a&
      &cc80(87)+acc80(88)+acc80(89)+acc80(90)+acc80(91)+acc80(92)+acc80(93)+acc&
      &80(94)+acc80(95)+acc80(96)+acc80(97)+acc80(98)+acc80(99)+acc80(100)+acc8&
      &0(101)+acc80(102)+acc80(103)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d80h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd80h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d80
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d80 = 0.0_ki
      d80 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d80, ki), aimag(d80), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d80h8l1_qp
