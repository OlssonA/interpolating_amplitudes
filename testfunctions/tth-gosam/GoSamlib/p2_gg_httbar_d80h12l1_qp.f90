module     p2_gg_httbar_d80h12l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d80h12l1_qp.f90
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
      use p2_gg_httbar_abbrevd80h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc80(109)
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspl4
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2l4
      complex(ki) :: QspQ
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak2k1
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspl4 = dotproduct(Q,l4)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      QspQ = dotproduct(Q,Q)
      Qspe1 = dotproduct(Q,e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
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
      acc80(15)=abb80(23)
      acc80(16)=abb80(24)
      acc80(17)=abb80(25)
      acc80(18)=abb80(26)
      acc80(19)=abb80(27)
      acc80(20)=abb80(28)
      acc80(21)=abb80(29)
      acc80(22)=abb80(30)
      acc80(23)=abb80(31)
      acc80(24)=abb80(32)
      acc80(25)=abb80(33)
      acc80(26)=abb80(34)
      acc80(27)=abb80(35)
      acc80(28)=abb80(36)
      acc80(29)=abb80(37)
      acc80(30)=abb80(41)
      acc80(31)=abb80(42)
      acc80(32)=abb80(43)
      acc80(33)=abb80(45)
      acc80(34)=abb80(46)
      acc80(35)=abb80(47)
      acc80(36)=abb80(50)
      acc80(37)=abb80(52)
      acc80(38)=abb80(53)
      acc80(39)=abb80(54)
      acc80(40)=abb80(56)
      acc80(41)=abb80(58)
      acc80(42)=abb80(60)
      acc80(43)=abb80(61)
      acc80(44)=abb80(62)
      acc80(45)=abb80(63)
      acc80(46)=abb80(66)
      acc80(47)=abb80(67)
      acc80(48)=abb80(69)
      acc80(49)=abb80(70)
      acc80(50)=abb80(71)
      acc80(51)=abb80(73)
      acc80(52)=abb80(74)
      acc80(53)=abb80(75)
      acc80(54)=abb80(76)
      acc80(55)=abb80(77)
      acc80(56)=abb80(80)
      acc80(57)=abb80(82)
      acc80(58)=abb80(83)
      acc80(59)=abb80(84)
      acc80(60)=abb80(85)
      acc80(61)=abb80(86)
      acc80(62)=abb80(87)
      acc80(63)=abb80(98)
      acc80(64)=abb80(106)
      acc80(65)=abb80(128)
      acc80(66)=abb80(134)
      acc80(67)=abb80(136)
      acc80(68)=abb80(137)
      acc80(69)=abb80(148)
      acc80(70)=abb80(149)
      acc80(71)=Qspvak2l3*acc80(23)
      acc80(72)=Qspval3l4*acc80(61)
      acc80(73)=Qspl4*acc80(17)
      acc80(74)=Qspvak2l5*acc80(22)
      acc80(75)=Qspval3k2*acc80(47)
      acc80(76)=Qspval3l5*acc80(56)
      acc80(77)=Qspval5l3*acc80(58)
      acc80(78)=Qspval5l4*acc80(37)
      acc80(79)=Qspvak2e2*acc80(10)
      acc80(80)=Qspval3e2*acc80(48)
      acc80(81)=Qspvae2l3*acc80(39)
      acc80(82)=Qspvae2l4*acc80(29)
      acc80(83)=Qspk2*acc80(8)
      acc80(84)=Qspvak2l4*acc80(16)
      acc80(85)=QspQ*acc80(15)
      acc80(71)=acc80(85)+acc80(84)+acc80(83)+acc80(82)+acc80(81)+acc80(80)+acc&
      &80(79)+acc80(78)+acc80(77)+acc80(76)+acc80(75)+acc80(74)+acc80(73)+acc80&
      &(72)+acc80(1)+acc80(71)
      acc80(71)=Qspe1*acc80(71)
      acc80(72)=-acc80(67)*Qspvae1l5
      acc80(73)=acc80(53)*Qspval5e1
      acc80(74)=acc80(43)*Qspvae2e1
      acc80(75)=acc80(30)*Qspvae1e2
      acc80(76)=-Qspvak2e1*acc80(64)
      acc80(77)=-Qspvae1k2*acc80(60)
      acc80(78)=Qspvae1l4*acc80(28)
      acc80(72)=acc80(78)+acc80(77)+acc80(76)+acc80(75)+acc80(74)+acc80(73)+acc&
      &80(5)+acc80(72)
      acc80(72)=QspQ*acc80(72)
      acc80(73)=-Qspvak2l5*acc80(51)
      acc80(74)=Qspvak2e2*acc80(14)
      acc80(75)=-Qspk2*acc80(41)
      acc80(76)=Qspvak2l4*acc80(21)
      acc80(73)=acc80(76)+acc80(75)+acc80(74)+acc80(2)+acc80(73)
      acc80(73)=Qspvae1k2*acc80(73)
      acc80(74)=Qspl4*acc80(6)
      acc80(75)=-acc80(67)*Qspval4l5
      acc80(76)=-acc80(60)*Qspval4k2
      acc80(77)=acc80(30)*Qspval4e2
      acc80(74)=acc80(77)+acc80(76)+acc80(75)+acc80(4)+acc80(74)
      acc80(74)=Qspvae1l4*acc80(74)
      acc80(75)=-acc80(67)*Qspval3l5
      acc80(76)=-acc80(60)*Qspval3k2
      acc80(77)=acc80(30)*Qspval3e2
      acc80(75)=acc80(77)+acc80(76)+acc80(49)+acc80(75)
      acc80(75)=Qspvae1l3*acc80(75)
      acc80(76)=acc80(53)*Qspval5l4
      acc80(77)=acc80(43)*Qspvae2l4
      acc80(78)=-Qspvak2l4*acc80(64)
      acc80(76)=acc80(78)+acc80(77)+acc80(31)+acc80(76)
      acc80(76)=Qspval4e1*acc80(76)
      acc80(77)=Qspval5k2*acc80(18)
      acc80(78)=Qspvae2k2*acc80(20)
      acc80(79)=-Qspk2*acc80(63)
      acc80(77)=acc80(79)+acc80(78)+acc80(3)+acc80(77)
      acc80(77)=Qspvak2e1*acc80(77)
      acc80(78)=acc80(53)*Qspval5l3
      acc80(79)=acc80(43)*Qspvae2l3
      acc80(78)=acc80(79)+acc80(78)+acc80(54)
      acc80(78)=Qspval3e1*acc80(78)
      acc80(79)=acc80(52)*Qspval3k1
      acc80(80)=acc80(50)*Qspvak1l3
      acc80(81)=acc80(35)*Qspvak1e1
      acc80(82)=acc80(34)*Qspvae1k1
      acc80(83)=acc80(26)*Qspvak1l4
      acc80(84)=acc80(24)*Qspvak1k2
      acc80(85)=acc80(19)*Qspval4k1
      acc80(86)=acc80(13)*Qspvak2k1
      acc80(87)=Qspvak2l3*acc80(7)
      acc80(88)=Qspval3l4*acc80(59)
      acc80(89)=Qspval4k2*acc80(45)
      acc80(90)=Qspval4l5*acc80(66)
      acc80(91)=-Qspval5k2*acc80(32)
      acc80(92)=Qspvae2k2*acc80(25)
      acc80(93)=-Qspval4e2*acc80(65)
      acc80(94)=Qspval5e1*acc80(40)
      acc80(95)=-Qspvae1l5*acc80(68)
      acc80(96)=Qspvae1e2*acc80(36)
      acc80(97)=Qspvae2e1*acc80(27)
      acc80(98)=Qspl4*acc80(9)
      acc80(99)=Qspvak2l5*acc80(55)
      acc80(100)=Qspval3k2*acc80(62)
      acc80(101)=Qspval3l5*acc80(42)
      acc80(102)=Qspval5l3*acc80(44)
      acc80(103)=-Qspval5l4*acc80(70)
      acc80(104)=Qspvak2e2*acc80(57)
      acc80(105)=Qspval3e2*acc80(46)
      acc80(106)=Qspvae2l3*acc80(38)
      acc80(107)=-Qspvae2l4*acc80(69)
      acc80(108)=Qspk2*acc80(33)
      acc80(109)=Qspvak2l4*acc80(12)
      brack=acc80(11)+acc80(71)+acc80(72)+acc80(73)+acc80(74)+acc80(75)+acc80(7&
      &6)+acc80(77)+acc80(78)+acc80(79)+acc80(80)+acc80(81)+acc80(82)+acc80(83)&
      &+acc80(84)+acc80(85)+acc80(86)+acc80(87)+acc80(88)+acc80(89)+acc80(90)+a&
      &cc80(91)+acc80(92)+acc80(93)+acc80(94)+acc80(95)+acc80(96)+acc80(97)+acc&
      &80(98)+acc80(99)+acc80(100)+acc80(101)+acc80(102)+acc80(103)+acc80(104)+&
      &acc80(105)+acc80(106)+acc80(107)+acc80(108)+acc80(109)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d80h12l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd80h12_qp
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
end module p2_gg_httbar_d80h12l1_qp
