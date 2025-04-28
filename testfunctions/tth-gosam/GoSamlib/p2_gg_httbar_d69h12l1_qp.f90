module     p2_gg_httbar_d69h12l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d69h12l1_qp.f90
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
      use p2_gg_httbar_abbrevd69h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc69(50)
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: QspQ
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspval3e2
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      QspQ = dotproduct(Q,Q)
      Qspk2 = dotproduct(Q,k2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      acc69(1)=abb69(9)
      acc69(2)=abb69(10)
      acc69(3)=abb69(11)
      acc69(4)=abb69(12)
      acc69(5)=abb69(13)
      acc69(6)=abb69(14)
      acc69(7)=abb69(15)
      acc69(8)=abb69(16)
      acc69(9)=abb69(17)
      acc69(10)=abb69(18)
      acc69(11)=abb69(19)
      acc69(12)=abb69(20)
      acc69(13)=abb69(21)
      acc69(14)=abb69(22)
      acc69(15)=abb69(23)
      acc69(16)=abb69(24)
      acc69(17)=abb69(25)
      acc69(18)=abb69(26)
      acc69(19)=abb69(27)
      acc69(20)=abb69(28)
      acc69(21)=abb69(29)
      acc69(22)=abb69(30)
      acc69(23)=abb69(31)
      acc69(24)=abb69(32)
      acc69(25)=abb69(33)
      acc69(26)=abb69(34)
      acc69(27)=abb69(35)
      acc69(28)=abb69(36)
      acc69(29)=abb69(37)
      acc69(30)=abb69(38)
      acc69(31)=abb69(40)
      acc69(32)=abb69(41)
      acc69(33)=abb69(43)
      acc69(34)=abb69(45)
      acc69(35)=abb69(52)
      acc69(36)=abb69(54)
      acc69(37)=abb69(55)
      acc69(38)=abb69(56)
      acc69(39)=acc69(6)*Qspvak1e2
      acc69(40)=acc69(10)*Qspvae2k2
      acc69(41)=acc69(18)*Qspvak2e2
      acc69(42)=-acc69(24)*Qspvae2l5
      acc69(43)=-acc69(26)*Qspvae2k1
      acc69(44)=-acc69(28)*Qspvae2e1
      acc69(45)=-acc69(34)*Qspvae1e2
      acc69(46)=acc69(36)*Qspvae2l4
      acc69(47)=-acc69(38)*Qspval4e2
      acc69(39)=acc69(19)+acc69(47)+acc69(46)+acc69(45)+acc69(44)+acc69(43)+acc&
      &69(42)+acc69(41)+acc69(39)+acc69(40)
      acc69(39)=QspQ*acc69(39)
      acc69(40)=acc69(8)*Qspvae2k2
      acc69(41)=acc69(12)*Qspk2
      acc69(42)=acc69(17)*Qspvae2e1
      acc69(43)=acc69(20)*Qspvae2k1
      acc69(44)=acc69(21)*Qspvae2l4
      acc69(45)=Qspvae1k2*acc69(22)
      acc69(46)=Qspval4k2*acc69(27)
      acc69(47)=Qspvak1k2*acc69(4)
      acc69(40)=acc69(47)+acc69(46)+acc69(45)+acc69(44)+acc69(43)+acc69(42)+acc&
      &69(41)+acc69(40)+acc69(1)
      acc69(40)=Qspvak2e2*acc69(40)
      acc69(41)=acc69(14)*Qspk2
      acc69(42)=Qspvak2e1*acc69(23)
      acc69(43)=Qspvak2l5*acc69(11)
      acc69(44)=Qspvak2l4*acc69(30)
      acc69(45)=Qspvak2k1*acc69(16)
      acc69(41)=acc69(45)+acc69(44)+acc69(43)+acc69(42)+acc69(41)+acc69(9)
      acc69(41)=Qspvae2k2*acc69(41)
      acc69(42)=Qspval3e1*acc69(28)
      acc69(43)=Qspval3l5*acc69(24)
      acc69(44)=-Qspval3l4*acc69(36)
      acc69(45)=-Qspval3k2*acc69(10)
      acc69(46)=Qspval3k1*acc69(26)
      acc69(42)=acc69(46)+acc69(45)+acc69(44)+acc69(43)+acc69(42)+acc69(2)
      acc69(42)=Qspvae2l3*acc69(42)
      acc69(43)=Qspvae1l3*acc69(34)
      acc69(44)=Qspval4l3*acc69(38)
      acc69(45)=-Qspvak2l3*acc69(18)
      acc69(46)=-Qspvak1l3*acc69(6)
      acc69(43)=acc69(46)+acc69(45)+acc69(44)+acc69(43)+acc69(3)
      acc69(43)=Qspval3e2*acc69(43)
      acc69(44)=acc69(13)*Qspvak1e2
      acc69(45)=acc69(31)*Qspval4e2
      acc69(46)=acc69(32)*Qspvae1e2
      acc69(44)=acc69(46)+acc69(45)+acc69(15)+acc69(44)
      acc69(44)=Qspvae2l5*acc69(44)
      acc69(45)=acc69(7)*Qspvak1e2
      acc69(46)=acc69(25)*Qspvae2k1
      acc69(47)=acc69(29)*Qspvae2e1
      acc69(48)=acc69(33)*Qspvae1e2
      acc69(49)=acc69(35)*Qspvae2l4
      acc69(50)=acc69(37)*Qspval4e2
      brack=acc69(5)+acc69(39)+acc69(40)+acc69(41)+acc69(42)+acc69(43)+acc69(44&
      &)+acc69(45)+acc69(46)+acc69(47)+acc69(48)+acc69(49)+acc69(50)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d69h12l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd69h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d69
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d69 = 0.0_ki
      d69 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d69, ki), aimag(d69), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d69h12l1_qp
