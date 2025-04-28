module     p2_gg_httbar_d68h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d68h0l1_qp.f90
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
      use p2_gg_httbar_abbrevd68h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc68(35)
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval5k2
      complex(ki) :: QspQ
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspl5
      complex(ki) :: Qspl4
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval5k2 = dotproduct(Q,spval5k2)
      QspQ = dotproduct(Q,Q)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspl5 = dotproduct(Q,l5)
      Qspl4 = dotproduct(Q,l4)
      acc68(1)=abb68(10)
      acc68(2)=abb68(11)
      acc68(3)=abb68(13)
      acc68(4)=abb68(14)
      acc68(5)=abb68(15)
      acc68(6)=abb68(16)
      acc68(7)=abb68(17)
      acc68(8)=abb68(18)
      acc68(9)=abb68(20)
      acc68(10)=abb68(22)
      acc68(11)=abb68(23)
      acc68(12)=abb68(24)
      acc68(13)=abb68(25)
      acc68(14)=abb68(26)
      acc68(15)=abb68(28)
      acc68(16)=abb68(29)
      acc68(17)=abb68(30)
      acc68(18)=abb68(32)
      acc68(19)=abb68(38)
      acc68(20)=abb68(45)
      acc68(21)=abb68(47)
      acc68(22)=abb68(48)
      acc68(23)=acc68(14)*Qspval4k1
      acc68(24)=acc68(15)*Qspval4k2
      acc68(25)=acc68(19)*Qspval5k1
      acc68(26)=acc68(20)*Qspval5k2
      acc68(23)=acc68(26)+acc68(25)+acc68(24)+acc68(23)+acc68(1)
      acc68(23)=QspQ*acc68(23)
      acc68(24)=acc68(10)*Qspval4k1
      acc68(25)=-acc68(21)*Qspval5k1
      acc68(24)=acc68(25)+acc68(24)+acc68(2)
      acc68(24)=Qspvak1k2*acc68(24)
      acc68(25)=acc68(13)*Qspval4k2
      acc68(26)=acc68(22)*Qspval5k2
      acc68(25)=acc68(26)+acc68(25)+acc68(3)
      acc68(25)=Qspk2*acc68(25)
      acc68(26)=acc68(7)*Qspval4k2
      acc68(27)=acc68(8)*Qspval4k1
      acc68(28)=acc68(16)*Qspval5k2
      acc68(29)=acc68(17)*Qspval5k1
      acc68(30)=Qspval5l4*acc68(18)
      acc68(31)=Qspval4l5*acc68(4)
      acc68(32)=Qspval4l3*acc68(12)
      acc68(33)=Qspval3k2*acc68(5)
      acc68(34)=Qspl5*acc68(9)
      acc68(35)=Qspl4*acc68(6)
      brack=acc68(11)+acc68(23)+acc68(24)+acc68(25)+acc68(26)+acc68(27)+acc68(2&
      &8)+acc68(29)+acc68(30)+acc68(31)+acc68(32)+acc68(33)+acc68(34)+acc68(35)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d68h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd68h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d68
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d68 = 0.0_ki
      d68 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d68, ki), aimag(d68), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d68h0l1_qp
