module     p2_gg_httbar_d195h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d195h4l1.f90
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
      use p2_gg_httbar_abbrevd195h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc195(58)
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspk1
      complex(ki) :: QspQ
      complex(ki) :: Qspe1
      complex(ki) :: Qspe2
      complex(ki) :: Qspk2
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspk1 = dotproduct(Q,k1)
      QspQ = dotproduct(Q,Q)
      Qspe1 = dotproduct(Q,e1)
      Qspe2 = dotproduct(Q,e2)
      Qspk2 = dotproduct(Q,k2)
      acc195(1)=abb195(72)
      acc195(2)=abb195(73)
      acc195(3)=abb195(74)
      acc195(4)=abb195(75)
      acc195(5)=abb195(76)
      acc195(6)=abb195(77)
      acc195(7)=abb195(78)
      acc195(8)=abb195(79)
      acc195(9)=abb195(80)
      acc195(10)=abb195(81)
      acc195(11)=abb195(83)
      acc195(12)=abb195(84)
      acc195(13)=abb195(85)
      acc195(14)=abb195(86)
      acc195(15)=abb195(87)
      acc195(16)=abb195(88)
      acc195(17)=abb195(89)
      acc195(18)=abb195(90)
      acc195(19)=abb195(91)
      acc195(20)=abb195(92)
      acc195(21)=abb195(93)
      acc195(22)=abb195(94)
      acc195(23)=abb195(95)
      acc195(24)=abb195(96)
      acc195(25)=abb195(97)
      acc195(26)=abb195(98)
      acc195(27)=abb195(99)
      acc195(28)=abb195(100)
      acc195(29)=abb195(101)
      acc195(30)=abb195(102)
      acc195(31)=abb195(103)
      acc195(32)=abb195(104)
      acc195(33)=abb195(105)
      acc195(34)=abb195(106)
      acc195(35)=abb195(107)
      acc195(36)=abb195(108)
      acc195(37)=abb195(109)
      acc195(38)=abb195(110)
      acc195(39)=abb195(111)
      acc195(40)=abb195(112)
      acc195(41)=abb195(113)
      acc195(42)=abb195(114)
      acc195(43)=abb195(115)
      acc195(44)=abb195(116)
      acc195(45)=abb195(150)
      acc195(46)=abb195(163)
      acc195(47)=-Qspvae2l3*acc195(19)
      acc195(48)=-Qspvae2l4*acc195(8)
      acc195(47)=acc195(48)+acc195(23)+acc195(47)
      acc195(47)=Qspval5e1*acc195(47)
      acc195(48)=Qspvak2e1*acc195(32)
      acc195(49)=Qspvak2e1*acc195(7)
      acc195(49)=acc195(34)+acc195(49)
      acc195(49)=Qspvae2k2*acc195(49)
      acc195(50)=Qspvae2k2*acc195(9)
      acc195(50)=acc195(17)+acc195(50)
      acc195(50)=Qspval3e1*acc195(50)
      acc195(51)=-Qspvak2e1*acc195(10)
      acc195(51)=acc195(35)+acc195(51)
      acc195(51)=Qspvae2l3*acc195(51)
      acc195(52)=Qspval3e1*acc195(13)
      acc195(52)=acc195(33)+acc195(52)
      acc195(52)=Qspvae2l4*acc195(52)
      acc195(47)=acc195(47)+acc195(52)+acc195(51)+acc195(50)+acc195(49)+acc195(&
      &27)+acc195(48)
      acc195(47)=Qspvae1e2*acc195(47)
      acc195(48)=-Qspvae1l3*acc195(19)
      acc195(49)=-Qspvae1l4*acc195(8)
      acc195(48)=acc195(49)+acc195(31)+acc195(48)
      acc195(48)=Qspval5e2*acc195(48)
      acc195(49)=Qspvae1k2*acc195(26)
      acc195(50)=Qspvae1k2*acc195(7)
      acc195(50)=acc195(30)+acc195(50)
      acc195(50)=Qspvak2e2*acc195(50)
      acc195(51)=-Qspvak2e2*acc195(10)
      acc195(51)=acc195(24)+acc195(51)
      acc195(51)=Qspvae1l3*acc195(51)
      acc195(52)=Qspvae1k2*acc195(9)
      acc195(52)=acc195(36)+acc195(52)
      acc195(52)=Qspval3e2*acc195(52)
      acc195(53)=Qspval3e2*acc195(13)
      acc195(53)=acc195(15)+acc195(53)
      acc195(53)=Qspvae1l4*acc195(53)
      acc195(48)=acc195(48)+acc195(53)+acc195(52)+acc195(51)+acc195(50)+acc195(&
      &20)+acc195(49)
      acc195(48)=Qspvae2e1*acc195(48)
      acc195(49)=Qspvak2l3*acc195(44)
      acc195(50)=Qspval3k2*acc195(42)
      acc195(51)=Qspval3l4*acc195(40)
      acc195(52)=Qspval5l3*acc195(38)
      acc195(53)=Qspval5l4*acc195(28)
      acc195(49)=acc195(53)-acc195(49)-acc195(50)+acc195(51)+acc195(52)
      acc195(50)=acc195(11)-acc195(49)
      acc195(50)=Qspk1*acc195(50)
      acc195(51)=acc195(5)+acc195(49)
      acc195(51)=QspQ*acc195(51)
      acc195(52)=Qspvak2l3*acc195(43)
      acc195(53)=Qspval3k2*acc195(41)
      acc195(54)=Qspval3l4*acc195(39)
      acc195(55)=Qspval5l3*acc195(37)
      acc195(56)=Qspval5l4*acc195(25)
      acc195(52)=acc195(56)+acc195(55)+acc195(54)+acc195(53)+acc195(2)+acc195(5&
      &2)
      acc195(52)=Qspe1*acc195(52)
      acc195(53)=QspQ*acc195(46)
      acc195(52)=acc195(52)+acc195(16)+acc195(53)
      acc195(52)=Qspe2*acc195(52)
      acc195(53)=QspQ-Qspk1
      acc195(54)=-Qspk2-acc195(53)
      acc195(54)=acc195(4)*acc195(54)
      acc195(55)=Qspe1*acc195(3)
      acc195(55)=acc195(46)+acc195(55)
      acc195(55)=Qspe2*acc195(55)
      acc195(49)=acc195(55)+acc195(21)+acc195(54)+acc195(49)
      acc195(49)=Qspk2*acc195(49)
      acc195(54)=Qspvak2l3*acc195(22)
      acc195(55)=Qspval3k2*acc195(29)
      acc195(56)=Qspval3l4*acc195(12)
      acc195(57)=Qspval5l3*acc195(18)
      acc195(58)=Qspval5l4*acc195(6)
      acc195(53)=acc195(45)*acc195(53)
      acc195(53)=acc195(1)+acc195(53)
      acc195(53)=Qspe1*acc195(53)
      brack=acc195(14)+acc195(47)+acc195(48)+acc195(49)+acc195(50)+acc195(51)+a&
      &cc195(52)+acc195(53)+acc195(54)+acc195(55)+acc195(56)+acc195(57)+acc195(&
      &58)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d195h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd195h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d195
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d195 = 0.0_ki
      d195 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d195, ki), aimag(d195), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d195h4l1
