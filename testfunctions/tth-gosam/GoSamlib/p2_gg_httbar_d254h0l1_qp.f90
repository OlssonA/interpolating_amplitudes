module     p2_gg_httbar_d254h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d254h0l1_qp.f90
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
      use p2_gg_httbar_abbrevd254h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc254(71)
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval5e1
      complex(ki) :: QspQ
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak1e2
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      QspQ = dotproduct(Q,Q)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      acc254(1)=abb254(7)
      acc254(2)=abb254(8)
      acc254(3)=abb254(9)
      acc254(4)=abb254(10)
      acc254(5)=abb254(11)
      acc254(6)=abb254(12)
      acc254(7)=abb254(13)
      acc254(8)=abb254(14)
      acc254(9)=abb254(15)
      acc254(10)=abb254(16)
      acc254(11)=abb254(17)
      acc254(12)=abb254(18)
      acc254(13)=abb254(19)
      acc254(14)=abb254(20)
      acc254(15)=abb254(21)
      acc254(16)=abb254(22)
      acc254(17)=abb254(23)
      acc254(18)=abb254(24)
      acc254(19)=abb254(25)
      acc254(20)=abb254(26)
      acc254(21)=abb254(27)
      acc254(22)=abb254(28)
      acc254(23)=abb254(29)
      acc254(24)=abb254(30)
      acc254(25)=abb254(31)
      acc254(26)=abb254(32)
      acc254(27)=abb254(33)
      acc254(28)=abb254(34)
      acc254(29)=abb254(35)
      acc254(30)=abb254(36)
      acc254(31)=abb254(37)
      acc254(32)=abb254(38)
      acc254(33)=abb254(39)
      acc254(34)=abb254(40)
      acc254(35)=abb254(41)
      acc254(36)=abb254(42)
      acc254(37)=abb254(44)
      acc254(38)=abb254(45)
      acc254(39)=abb254(46)
      acc254(40)=abb254(47)
      acc254(41)=abb254(48)
      acc254(42)=abb254(50)
      acc254(43)=abb254(51)
      acc254(44)=abb254(52)
      acc254(45)=abb254(53)
      acc254(46)=abb254(54)
      acc254(47)=abb254(55)
      acc254(48)=abb254(56)
      acc254(49)=abb254(57)
      acc254(50)=abb254(58)
      acc254(51)=abb254(59)
      acc254(52)=abb254(60)
      acc254(53)=Qspval4k2*acc254(23)
      acc254(54)=Qspval5k2*acc254(49)
      acc254(53)=acc254(53)+acc254(54)
      acc254(53)=Qspvak2e1*acc254(53)
      acc254(54)=acc254(35)*Qspval4k1
      acc254(55)=-Qspval5k1*acc254(1)
      acc254(54)=acc254(55)+acc254(33)+acc254(54)
      acc254(54)=Qspvak1e1*acc254(54)
      acc254(55)=-acc254(35)*Qspval4l3
      acc254(56)=acc254(1)*Qspval5l3
      acc254(55)=acc254(56)+acc254(45)+acc254(55)
      acc254(55)=Qspval3e1*acc254(55)
      acc254(56)=-acc254(35)*Qspval4e1
      acc254(57)=Qspval5e1*acc254(1)
      acc254(56)=acc254(57)+acc254(39)+acc254(56)
      acc254(56)=QspQ*acc254(56)
      acc254(57)=Qspvak1k2*acc254(25)
      acc254(58)=Qspval4k2*acc254(31)
      acc254(59)=Qspval4l3*acc254(51)
      acc254(60)=Qspval5k2*acc254(40)
      acc254(61)=Qspval5l3*acc254(47)
      acc254(62)=Qspval4e1*acc254(10)
      acc254(63)=Qspval5e1*acc254(42)
      acc254(64)=Qspval5e1*acc254(36)
      acc254(64)=acc254(13)+acc254(64)
      acc254(64)=Qspvae2k2*acc254(64)
      acc254(53)=acc254(64)+acc254(56)+acc254(55)+acc254(54)+acc254(63)+acc254(&
      &62)+acc254(53)+acc254(61)+acc254(60)+acc254(59)+acc254(58)+acc254(4)+acc&
      &254(57)
      acc254(53)=Qspvae1e2*acc254(53)
      acc254(54)=Qspvae1k1*acc254(15)
      acc254(55)=Qspvae1l3*acc254(43)
      acc254(56)=Qspval5k1*acc254(28)
      acc254(57)=Qspvae1k2*acc254(21)
      acc254(58)=QspQ*acc254(26)
      acc254(59)=Qspvae1k2*acc254(3)
      acc254(59)=acc254(9)+acc254(59)
      acc254(59)=Qspval4e2*acc254(59)
      acc254(54)=acc254(59)+acc254(58)+acc254(57)+acc254(56)+acc254(55)+acc254(&
      &16)+acc254(54)
      acc254(54)=Qspvae2e1*acc254(54)
      acc254(55)=Qspvae1k1*acc254(32)
      acc254(56)=Qspvae1l3*acc254(37)
      acc254(57)=Qspval5k1*acc254(48)
      acc254(58)=Qspvae1k2*acc254(24)
      acc254(59)=QspQ*acc254(14)
      acc254(55)=acc254(59)+acc254(58)+acc254(57)+acc254(56)+acc254(2)+acc254(5&
      &5)
      acc254(55)=Qspvae2k2*acc254(55)
      acc254(56)=Qspvak1k2*acc254(19)
      acc254(57)=Qspvak1e1*acc254(34)
      acc254(58)=Qspval3e1*acc254(46)
      acc254(59)=QspQ*acc254(29)
      acc254(56)=acc254(59)+acc254(58)+acc254(57)+acc254(20)+acc254(56)
      acc254(56)=Qspval4e2*acc254(56)
      acc254(57)=acc254(27)*Qspvak1e2
      acc254(58)=Qspval4k1*acc254(11)
      acc254(59)=Qspvak1k2*acc254(22)
      acc254(60)=Qspval4k2*acc254(52)
      acc254(61)=Qspval4l3*acc254(50)
      acc254(62)=Qspval5k2*acc254(5)
      acc254(63)=Qspval5l3*acc254(38)
      acc254(64)=Qspvak2e1*acc254(30)
      acc254(65)=Qspval4e1*acc254(41)
      acc254(66)=Qspval5k1*acc254(8)
      acc254(67)=Qspvae1k2*acc254(6)
      acc254(68)=Qspval5e1*acc254(17)
      acc254(69)=Qspvak1e1*acc254(7)
      acc254(70)=Qspval3e1*acc254(44)
      acc254(71)=QspQ*acc254(18)
      brack=acc254(12)+acc254(53)+acc254(54)+acc254(55)+acc254(56)+acc254(57)+a&
      &cc254(58)+acc254(59)+acc254(60)+acc254(61)+acc254(62)+acc254(63)+acc254(&
      &64)+acc254(65)+acc254(66)+acc254(67)+acc254(68)+acc254(69)+acc254(70)+ac&
      &c254(71)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d254h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd254h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d254
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d254 = 0.0_ki
      d254 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d254, ki), aimag(d254), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d254h0l1_qp
