module     p2_gg_httbar_d29h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d29h4l1.f90
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
      use p2_gg_httbar_abbrevd29h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc29(73)
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc29(1)=abb29(11)
      acc29(2)=abb29(12)
      acc29(3)=abb29(13)
      acc29(4)=abb29(14)
      acc29(5)=abb29(15)
      acc29(6)=abb29(16)
      acc29(7)=abb29(17)
      acc29(8)=abb29(18)
      acc29(9)=abb29(19)
      acc29(10)=abb29(20)
      acc29(11)=abb29(21)
      acc29(12)=abb29(22)
      acc29(13)=abb29(23)
      acc29(14)=abb29(24)
      acc29(15)=abb29(25)
      acc29(16)=abb29(26)
      acc29(17)=abb29(27)
      acc29(18)=abb29(28)
      acc29(19)=abb29(29)
      acc29(20)=abb29(30)
      acc29(21)=abb29(31)
      acc29(22)=abb29(32)
      acc29(23)=abb29(33)
      acc29(24)=abb29(34)
      acc29(25)=abb29(35)
      acc29(26)=abb29(37)
      acc29(27)=abb29(39)
      acc29(28)=abb29(40)
      acc29(29)=abb29(42)
      acc29(30)=abb29(43)
      acc29(31)=abb29(44)
      acc29(32)=abb29(45)
      acc29(33)=abb29(49)
      acc29(34)=abb29(52)
      acc29(35)=abb29(57)
      acc29(36)=abb29(72)
      acc29(37)=abb29(73)
      acc29(38)=Qspvae2e1*acc29(34)
      acc29(39)=Qspvae1e2*acc29(12)
      acc29(40)=Qspvae2l5*acc29(13)
      acc29(41)=Qspval5e2*acc29(20)
      acc29(42)=Qspvae1l4*acc29(21)
      acc29(43)=Qspval4e1*acc29(25)
      acc29(44)=Qspvae2l3*acc29(27)
      acc29(45)=Qspval3e2*acc29(29)
      acc29(46)=Qspvae1l3*acc29(37)
      acc29(47)=-Qspval3e1*acc29(35)
      acc29(48)=Qspvae2k2*acc29(30)
      acc29(49)=Qspvak2e2*acc29(1)
      acc29(50)=Qspvae1k2*acc29(2)
      acc29(51)=-Qspvak2e1*acc29(36)
      acc29(52)=Qspvae2k1*acc29(9)
      acc29(53)=Qspvak1e2*acc29(16)
      acc29(54)=Qspval5l4*acc29(24)
      acc29(55)=Qspval5l3*acc29(28)
      acc29(56)=Qspval5k2*acc29(23)
      acc29(57)=Qspval4l5*acc29(31)
      acc29(58)=Qspval4l3*acc29(32)
      acc29(59)=Qspval4k2*acc29(33)
      acc29(60)=Qspval4k1*acc29(26)
      acc29(61)=Qspval3l5*acc29(10)
      acc29(62)=Qspval3l4*acc29(11)
      acc29(63)=Qspval3k2*acc29(5)
      acc29(64)=Qspval3k1*acc29(8)
      acc29(65)=Qspvak2l5*acc29(15)
      acc29(66)=Qspvak2l4*acc29(6)
      acc29(67)=Qspvak2l3*acc29(14)
      acc29(68)=Qspvak2k1*acc29(17)
      acc29(69)=Qspvak1l4*acc29(19)
      acc29(70)=Qspvak1l3*acc29(22)
      acc29(71)=Qspvak1k2*acc29(18)
      acc29(72)=Qspk2*acc29(4)
      acc29(73)=QspQ*acc29(7)
      brack=acc29(3)+acc29(38)+acc29(39)+acc29(40)+acc29(41)+acc29(42)+acc29(43&
      &)+acc29(44)+acc29(45)+acc29(46)+acc29(47)+acc29(48)+acc29(49)+acc29(50)+&
      &acc29(51)+acc29(52)+acc29(53)+acc29(54)+acc29(55)+acc29(56)+acc29(57)+ac&
      &c29(58)+acc29(59)+acc29(60)+acc29(61)+acc29(62)+acc29(63)+acc29(64)+acc2&
      &9(65)+acc29(66)+acc29(67)+acc29(68)+acc29(69)+acc29(70)+acc29(71)+acc29(&
      &72)+acc29(73)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d29h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd29h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d29
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d29 = 0.0_ki
      d29 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d29, ki), aimag(d29), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d29h4l1
