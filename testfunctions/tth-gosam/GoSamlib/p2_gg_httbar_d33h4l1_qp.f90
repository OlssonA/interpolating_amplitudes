module     p2_gg_httbar_d33h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d33h4l1_qp.f90
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
      use p2_gg_httbar_abbrevd33h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc33(63)
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspl4
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspe1 = dotproduct(Q,e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspl4 = dotproduct(Q,l4)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc33(1)=abb33(9)
      acc33(2)=abb33(10)
      acc33(3)=abb33(11)
      acc33(4)=abb33(12)
      acc33(5)=abb33(13)
      acc33(6)=abb33(14)
      acc33(7)=abb33(15)
      acc33(8)=abb33(16)
      acc33(9)=abb33(17)
      acc33(10)=abb33(18)
      acc33(11)=abb33(19)
      acc33(12)=abb33(20)
      acc33(13)=abb33(21)
      acc33(14)=abb33(22)
      acc33(15)=abb33(23)
      acc33(16)=abb33(24)
      acc33(17)=abb33(25)
      acc33(18)=abb33(26)
      acc33(19)=abb33(27)
      acc33(20)=abb33(28)
      acc33(21)=abb33(29)
      acc33(22)=abb33(30)
      acc33(23)=abb33(31)
      acc33(24)=abb33(32)
      acc33(25)=abb33(33)
      acc33(26)=abb33(35)
      acc33(27)=abb33(36)
      acc33(28)=abb33(37)
      acc33(29)=abb33(38)
      acc33(30)=abb33(39)
      acc33(31)=abb33(40)
      acc33(32)=abb33(41)
      acc33(33)=abb33(42)
      acc33(34)=abb33(43)
      acc33(35)=abb33(44)
      acc33(36)=abb33(45)
      acc33(37)=abb33(46)
      acc33(38)=acc33(13)*Qspvae2k2
      acc33(39)=acc33(14)*Qspval5l4
      acc33(40)=acc33(25)*Qspval5k2
      acc33(41)=acc33(27)*Qspval3l4
      acc33(42)=acc33(32)*Qspval4l3
      acc33(43)=acc33(34)*Qspval3k2
      acc33(44)=acc33(37)*Qspval4k2
      acc33(45)=Qspval5e2*acc33(15)
      acc33(46)=Qspvae2l3*acc33(33)
      acc33(47)=Qspval3e2*acc33(20)
      acc33(38)=acc33(47)+acc33(46)+acc33(45)+acc33(44)+acc33(43)+acc33(42)+acc&
      &33(41)+acc33(40)+acc33(39)+acc33(38)+acc33(7)
      acc33(38)=Qspe1*acc33(38)
      acc33(39)=acc33(5)*Qspvae2k2
      acc33(40)=acc33(19)*Qspval5l4
      acc33(41)=acc33(22)*Qspval5k2
      acc33(42)=acc33(23)*Qspval4l3
      acc33(43)=acc33(26)*Qspval3k2
      acc33(44)=acc33(28)*Qspval4k2
      acc33(45)=acc33(29)*Qspval3l4
      acc33(46)=Qspvae2e1*acc33(10)
      acc33(47)=Qspvae1e2*acc33(1)
      acc33(48)=Qspval5e1*acc33(16)
      acc33(49)=Qspvae2l4*acc33(24)
      acc33(50)=Qspval4e2*acc33(30)
      acc33(51)=Qspvae1l4*acc33(21)
      acc33(52)=Qspval4e1*acc33(31)
      acc33(53)=Qspvae1l3*acc33(18)
      acc33(54)=Qspval3e1*acc33(35)
      acc33(55)=Qspvak2e2*acc33(4)
      acc33(56)=Qspvae1k2*acc33(6)
      acc33(57)=Qspvae1k1*acc33(12)
      acc33(58)=Qspvak1e1*acc33(3)
      acc33(59)=Qspvak2l4*acc33(11)
      acc33(60)=Qspvak2l3*acc33(36)
      acc33(61)=Qspl4*acc33(8)
      acc33(62)=Qspk2*acc33(9)
      acc33(63)=-QspQ*acc33(17)
      brack=acc33(2)+acc33(38)+acc33(39)+acc33(40)+acc33(41)+acc33(42)+acc33(43&
      &)+acc33(44)+acc33(45)+acc33(46)+acc33(47)+acc33(48)+acc33(49)+acc33(50)+&
      &acc33(51)+acc33(52)+acc33(53)+acc33(54)+acc33(55)+acc33(56)+acc33(57)+ac&
      &c33(58)+acc33(59)+acc33(60)+acc33(61)+acc33(62)+acc33(63)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d33h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd33h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d33
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d33 = 0.0_ki
      d33 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d33, ki), aimag(d33), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d33h4l1_qp
