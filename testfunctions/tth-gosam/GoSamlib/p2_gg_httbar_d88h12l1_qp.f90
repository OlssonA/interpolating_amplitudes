module     p2_gg_httbar_d88h12l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d88h12l1_qp.f90
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
      use p2_gg_httbar_abbrevd88h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc88(105)
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspe1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspl4
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk1
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspe1 = dotproduct(Q,e1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspl4 = dotproduct(Q,l4)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      Qspe2 = dotproduct(Q,e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk1 = dotproduct(Q,k1)
      acc88(1)=abb88(8)
      acc88(2)=abb88(9)
      acc88(3)=abb88(10)
      acc88(4)=abb88(11)
      acc88(5)=abb88(12)
      acc88(6)=abb88(13)
      acc88(7)=abb88(14)
      acc88(8)=abb88(15)
      acc88(9)=abb88(16)
      acc88(10)=abb88(17)
      acc88(11)=abb88(18)
      acc88(12)=abb88(19)
      acc88(13)=abb88(20)
      acc88(14)=abb88(21)
      acc88(15)=abb88(22)
      acc88(16)=abb88(23)
      acc88(17)=abb88(24)
      acc88(18)=abb88(25)
      acc88(19)=abb88(26)
      acc88(20)=abb88(27)
      acc88(21)=abb88(28)
      acc88(22)=abb88(29)
      acc88(23)=abb88(30)
      acc88(24)=abb88(31)
      acc88(25)=abb88(32)
      acc88(26)=abb88(33)
      acc88(27)=abb88(34)
      acc88(28)=abb88(35)
      acc88(29)=abb88(36)
      acc88(30)=abb88(37)
      acc88(31)=abb88(38)
      acc88(32)=abb88(39)
      acc88(33)=abb88(40)
      acc88(34)=abb88(41)
      acc88(35)=abb88(42)
      acc88(36)=abb88(43)
      acc88(37)=abb88(44)
      acc88(38)=abb88(45)
      acc88(39)=abb88(46)
      acc88(40)=abb88(47)
      acc88(41)=abb88(48)
      acc88(42)=abb88(49)
      acc88(43)=abb88(50)
      acc88(44)=abb88(51)
      acc88(45)=abb88(52)
      acc88(46)=abb88(53)
      acc88(47)=abb88(54)
      acc88(48)=abb88(55)
      acc88(49)=abb88(56)
      acc88(50)=abb88(57)
      acc88(51)=abb88(58)
      acc88(52)=abb88(59)
      acc88(53)=abb88(60)
      acc88(54)=abb88(61)
      acc88(55)=abb88(62)
      acc88(56)=abb88(63)
      acc88(57)=abb88(64)
      acc88(58)=abb88(65)
      acc88(59)=abb88(67)
      acc88(60)=abb88(69)
      acc88(61)=abb88(70)
      acc88(62)=abb88(71)
      acc88(63)=abb88(72)
      acc88(64)=abb88(73)
      acc88(65)=abb88(74)
      acc88(66)=abb88(75)
      acc88(67)=abb88(76)
      acc88(68)=abb88(77)
      acc88(69)=abb88(78)
      acc88(70)=abb88(79)
      acc88(71)=abb88(81)
      acc88(72)=abb88(82)
      acc88(73)=abb88(84)
      acc88(74)=abb88(85)
      acc88(75)=abb88(86)
      acc88(76)=abb88(87)
      acc88(77)=abb88(88)
      acc88(78)=abb88(89)
      acc88(79)=abb88(90)
      acc88(80)=abb88(92)
      acc88(81)=abb88(93)
      acc88(82)=abb88(94)
      acc88(83)=abb88(96)
      acc88(84)=Qspvak2l3*acc88(82)
      acc88(85)=Qspvak2l5*acc88(79)
      acc88(86)=Qspval3l4*acc88(75)
      acc88(87)=Qspvak2l4*acc88(9)
      acc88(84)=acc88(87)+acc88(86)+acc88(85)+acc88(2)+acc88(84)
      acc88(84)=Qspe1*acc88(84)
      acc88(85)=acc88(64)*Qspvak1e1
      acc88(86)=acc88(58)*Qspvae1k1
      acc88(87)=acc88(35)*Qspvak2e1
      acc88(88)=acc88(32)*Qspvae1l4
      acc88(89)=acc88(28)*Qspvae1l5
      acc88(90)=-acc88(26)*Qspvae1k2
      acc88(91)=acc88(24)*Qspvae1l3
      acc88(92)=acc88(23)*Qspval3e1
      acc88(93)=Qspvak1l3*acc88(37)
      acc88(94)=Qspvak1l5*acc88(42)
      acc88(95)=Qspval3k1*acc88(83)
      acc88(96)=Qspval3k2*acc88(22)
      acc88(97)=Qspval4l3*acc88(72)
      acc88(98)=Qspval4l5*acc88(66)
      acc88(99)=Qspl4*acc88(47)
      acc88(100)=Qspvak1l4*acc88(49)
      acc88(101)=Qspvak2k1*acc88(16)
      acc88(102)=Qspval3l4*acc88(67)
      acc88(103)=Qspk2*acc88(56)
      acc88(104)=QspQ*acc88(3)
      acc88(105)=Qspvak2l4*acc88(63)
      acc88(84)=acc88(84)+acc88(105)+acc88(104)+acc88(103)+acc88(102)+acc88(101&
      &)+acc88(100)+acc88(99)+acc88(98)+acc88(97)+acc88(96)+acc88(95)+acc88(94)&
      &+acc88(93)+acc88(92)+acc88(91)+acc88(90)+acc88(89)+acc88(88)+acc88(87)+a&
      &cc88(48)+acc88(85)+acc88(86)
      acc88(84)=Qspe2*acc88(84)
      acc88(85)=acc88(62)*Qspvae2k2
      acc88(86)=acc88(45)*Qspval3e2
      acc88(87)=acc88(41)*Qspvae2l3
      acc88(88)=acc88(36)*Qspvae2k1
      acc88(89)=acc88(30)*Qspvae2l4
      acc88(90)=acc88(17)*Qspvae2l5
      acc88(91)=acc88(12)*Qspvak1e2
      acc88(92)=acc88(11)*Qspvak2e2
      acc88(93)=Qspvak1l3*acc88(21)
      acc88(94)=Qspvak1l5*acc88(18)
      acc88(95)=Qspval3k1*acc88(60)
      acc88(96)=Qspval3k2*acc88(78)
      acc88(97)=Qspval4l3*acc88(73)
      acc88(98)=Qspval4l5*acc88(69)
      acc88(99)=-Qspl4*acc88(7)
      acc88(100)=Qspvak1l4*acc88(52)
      acc88(101)=Qspvak2k1*acc88(19)
      acc88(102)=Qspval3l4*acc88(74)
      acc88(103)=Qspk2*acc88(6)
      acc88(104)=QspQ*acc88(5)
      acc88(105)=Qspvak2l4*acc88(40)
      acc88(85)=acc88(105)+acc88(104)+acc88(103)+acc88(102)+acc88(101)+acc88(10&
      &0)+acc88(99)+acc88(98)+acc88(97)+acc88(96)+acc88(95)+acc88(94)+acc88(93)&
      &+acc88(92)+acc88(91)+acc88(90)+acc88(89)+acc88(88)+acc88(38)+acc88(87)+a&
      &cc88(85)+acc88(86)
      acc88(85)=Qspe1*acc88(85)
      acc88(86)=Qspval3k2*acc88(39)
      acc88(87)=Qspval4l3*acc88(81)
      acc88(88)=Qspval4l5*acc88(80)
      acc88(89)=Qspl4*acc88(70)
      acc88(90)=Qspk2*acc88(31)
      acc88(91)=QspQ*acc88(76)
      acc88(86)=acc88(91)+acc88(90)+acc88(89)+acc88(88)+acc88(87)+acc88(46)+acc&
      &88(86)
      acc88(86)=Qspvak2l4*acc88(86)
      acc88(87)=Qspvak2l3*acc88(81)
      acc88(88)=Qspvak2l5*acc88(80)
      acc88(89)=Qspvae1e2*acc88(15)
      acc88(90)=Qspvae2e1*acc88(10)
      acc88(91)=Qspval3l4*acc88(53)
      acc88(87)=acc88(91)+acc88(90)+acc88(89)+acc88(88)+acc88(1)+acc88(87)
      acc88(87)=QspQ*acc88(87)
      acc88(88)=Qspvak2l3*acc88(51)
      acc88(89)=Qspvak2l5*acc88(57)
      acc88(90)=Qspvae1e2*acc88(13)
      acc88(91)=Qspvae2e1*acc88(25)
      acc88(88)=acc88(91)+acc88(90)+acc88(89)+acc88(34)+acc88(88)
      acc88(88)=Qspk2*acc88(88)
      acc88(89)=-Qspvak1l3*acc88(81)
      acc88(90)=-Qspvak1l5*acc88(80)
      acc88(91)=-Qspvak1l4*acc88(76)
      acc88(89)=acc88(91)+acc88(90)+acc88(4)+acc88(89)
      acc88(89)=Qspvak2k1*acc88(89)
      acc88(90)=acc88(54)*Qspvak1k2
      acc88(91)=-Qspk1*acc88(27)
      acc88(92)=Qspvak1l3*acc88(33)
      acc88(93)=Qspvak1l5*acc88(29)
      acc88(94)=Qspvak2l3*acc88(61)
      acc88(95)=Qspvak2l5*acc88(59)
      acc88(96)=Qspval3k1*acc88(68)
      acc88(97)=Qspval3k2*acc88(50)
      acc88(98)=Qspval4l3*acc88(71)
      acc88(99)=Qspval4l5*acc88(65)
      acc88(100)=-Qspk1*acc88(13)
      acc88(100)=acc88(14)+acc88(100)
      acc88(100)=Qspvae1e2*acc88(100)
      acc88(101)=-Qspk1*acc88(25)
      acc88(101)=acc88(8)+acc88(101)
      acc88(101)=Qspvae2e1*acc88(101)
      acc88(102)=Qspl4*acc88(20)
      acc88(103)=-Qspval3k1*acc88(53)
      acc88(103)=acc88(43)+acc88(103)
      acc88(103)=Qspvak1l4*acc88(103)
      acc88(104)=Qspl4*acc88(77)
      acc88(104)=acc88(55)+acc88(104)
      acc88(104)=Qspval3l4*acc88(104)
      brack=acc88(44)+acc88(84)+acc88(85)+acc88(86)+acc88(87)+acc88(88)+acc88(8&
      &9)+acc88(90)+acc88(91)+acc88(92)+acc88(93)+acc88(94)+acc88(95)+acc88(96)&
      &+acc88(97)+acc88(98)+acc88(99)+acc88(100)+acc88(101)+acc88(102)+acc88(10&
      &3)+acc88(104)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d88h12l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd88h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d88
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d88 = 0.0_ki
      d88 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d88, ki), aimag(d88), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d88h12l1_qp
