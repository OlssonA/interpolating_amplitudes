module     p2_gg_httbar_d35h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d35h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd35h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc35(63)
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspl5
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspe1 = dotproduct(Q,e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspl5 = dotproduct(Q,l5)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc35(1)=abb35(9)
      acc35(2)=abb35(10)
      acc35(3)=abb35(11)
      acc35(4)=abb35(12)
      acc35(5)=abb35(13)
      acc35(6)=abb35(14)
      acc35(7)=abb35(15)
      acc35(8)=abb35(16)
      acc35(9)=abb35(17)
      acc35(10)=abb35(18)
      acc35(11)=abb35(19)
      acc35(12)=abb35(20)
      acc35(13)=abb35(21)
      acc35(14)=abb35(22)
      acc35(15)=abb35(23)
      acc35(16)=abb35(24)
      acc35(17)=abb35(25)
      acc35(18)=abb35(26)
      acc35(19)=abb35(27)
      acc35(20)=abb35(28)
      acc35(21)=abb35(29)
      acc35(22)=abb35(30)
      acc35(23)=abb35(31)
      acc35(24)=abb35(32)
      acc35(25)=abb35(33)
      acc35(26)=abb35(35)
      acc35(27)=abb35(36)
      acc35(28)=abb35(37)
      acc35(29)=abb35(38)
      acc35(30)=abb35(39)
      acc35(31)=abb35(40)
      acc35(32)=abb35(41)
      acc35(33)=abb35(42)
      acc35(34)=abb35(43)
      acc35(35)=abb35(44)
      acc35(36)=abb35(45)
      acc35(37)=abb35(46)
      acc35(38)=acc35(13)*Qspvae2k2
      acc35(39)=acc35(14)*Qspval4l5
      acc35(40)=acc35(22)*Qspval5l3
      acc35(41)=acc35(25)*Qspval5k2
      acc35(42)=acc35(27)*Qspval3l5
      acc35(43)=acc35(34)*Qspval3k2
      acc35(44)=acc35(37)*Qspval4k2
      acc35(45)=Qspval4e2*acc35(24)
      acc35(46)=Qspvae2l3*acc35(33)
      acc35(47)=Qspval3e2*acc35(20)
      acc35(38)=acc35(47)+acc35(46)+acc35(45)+acc35(44)+acc35(43)+acc35(42)+acc&
      &35(41)+acc35(40)+acc35(39)+acc35(38)+acc35(7)
      acc35(38)=Qspe1*acc35(38)
      acc35(39)=acc35(5)*Qspvae2k2
      acc35(40)=acc35(15)*Qspval5l3
      acc35(41)=acc35(23)*Qspval5k2
      acc35(42)=acc35(26)*Qspval3k2
      acc35(43)=acc35(28)*Qspval4l5
      acc35(44)=acc35(29)*Qspval3l5
      acc35(45)=acc35(32)*Qspval4k2
      acc35(46)=Qspvae2e1*acc35(10)
      acc35(47)=Qspvae1e2*acc35(1)
      acc35(48)=Qspvae2l5*acc35(19)
      acc35(49)=Qspval5e2*acc35(21)
      acc35(50)=Qspvae1l5*acc35(16)
      acc35(51)=Qspval5e1*acc35(30)
      acc35(52)=Qspval4e1*acc35(31)
      acc35(53)=Qspvae1l3*acc35(18)
      acc35(54)=Qspval3e1*acc35(35)
      acc35(55)=Qspvak2e2*acc35(4)
      acc35(56)=Qspvae1k2*acc35(6)
      acc35(57)=Qspvae1k1*acc35(12)
      acc35(58)=Qspvak1e1*acc35(3)
      acc35(59)=Qspvak2l5*acc35(11)
      acc35(60)=Qspvak2l3*acc35(36)
      acc35(61)=Qspl5*acc35(8)
      acc35(62)=Qspk2*acc35(9)
      acc35(63)=QspQ*acc35(17)
      brack=acc35(2)+acc35(38)+acc35(39)+acc35(40)+acc35(41)+acc35(42)+acc35(43&
      &)+acc35(44)+acc35(45)+acc35(46)+acc35(47)+acc35(48)+acc35(49)+acc35(50)+&
      &acc35(51)+acc35(52)+acc35(53)+acc35(54)+acc35(55)+acc35(56)+acc35(57)+ac&
      &c35(58)+acc35(59)+acc35(60)+acc35(61)+acc35(62)+acc35(63)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d35h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd35h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d35
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d35 = 0.0_ki
      d35 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d35, ki), aimag(d35), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d35h8l1_qp
